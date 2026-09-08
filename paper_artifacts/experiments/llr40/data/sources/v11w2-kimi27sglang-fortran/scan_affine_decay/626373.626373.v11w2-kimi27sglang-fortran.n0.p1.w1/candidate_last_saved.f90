subroutine scan_affine_decay_fp64(c, x, y, LEN_1D, workspace, workspace_bytes) bind(c, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_ptr, c_f_pointer, c_associated
  use omp_lib, only: omp_get_max_threads
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D, workspace_bytes
  real(c_double), intent(in) :: c(LEN_1D), x(LEN_1D)
  real(c_double), intent(inout) :: y(LEN_1D)
  type(c_ptr), intent(in), value :: workspace

  integer(c_int64_t), parameter :: BLK = 4096
  integer(c_int64_t) :: n, nchunks, ch, s, e, i
  real(c_double), pointer :: blk_c(:), blk_x(:), init(:)
  real(c_double), allocatable, target :: tmp_c(:), tmp_x(:), tmp_i(:)
  real(c_double) :: cc, xx, carry
  logical :: use_scratch

  n = LEN_1D
  if (n <= 1) return

  if (n < BLK .or. omp_get_max_threads() == 1) then
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  nchunks = (n + BLK - 1) / BLK

  use_scratch = c_associated(workspace) .and. (workspace_bytes >= 24_c_int64_t * nchunks)
  if (use_scratch) then
    block
      real(c_double), pointer :: work(:)
      call c_f_pointer(workspace, work, [3 * nchunks])
      blk_c => work(1:nchunks)
      blk_x => work(nchunks + 1:2 * nchunks)
      init => work(2 * nchunks + 1:3 * nchunks)
    end block
  else
    allocate(tmp_c(nchunks), tmp_x(nchunks), tmp_i(nchunks))
    blk_c => tmp_c
    blk_x => tmp_x
    init => tmp_i
  end if

  !$omp parallel do schedule(static) private(s, e, i, cc, xx)
  do ch = 1, nchunks
    s = (ch - 1) * BLK + 1
    e = min(ch * BLK, n)
    cc = c(s)
    xx = x(s)
    do i = s + 1, e
      cc = cc * c(i)
      xx = c(i) * xx + x(i)
    end do
    blk_c(ch) = cc
    blk_x(ch) = xx
  end do

  carry = 0.0_c_double
  do ch = 1, nchunks
    init(ch) = carry
    carry = blk_c(ch) * carry + blk_x(ch)
  end do

  !$omp parallel do schedule(static) private(s, e, i, carry)
  do ch = 1, nchunks
    s = (ch - 1) * BLK + 1
    e = min(ch * BLK, n)
    carry = init(ch)
    do i = s, e
      carry = c(i) * carry + x(i)
      y(i) = carry
    end do
  end do

  if (.not. use_scratch) then
    deallocate(tmp_c, tmp_x, tmp_i)
  end if
end subroutine scan_affine_decay_fp64
