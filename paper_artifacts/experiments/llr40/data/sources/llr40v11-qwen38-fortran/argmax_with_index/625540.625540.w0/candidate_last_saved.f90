module amwi_mod
  use iso_c_binding
  implicit none
  integer(c_int64_t), parameter :: BS = 4096_c_int64_t
  real(c_double), allocatable :: bm(:)
  integer(c_int64_t) :: bm_cap = 0_c_int64_t
  public
contains
  subroutine amwi_grow(nb)
    integer(c_int64_t), intent(in) :: nb
    if (nb > bm_cap) then
      if (allocated(bm)) deallocate(bm)
      allocate(bm(nb))
      bm_cap = nb
    end if
  end subroutine amwi_grow

  ! Exact real equality via bit pattern; -0.0 and +0.0 compare equal,
  ! NaN never equals anything (including itself).
  function amwi_equal(x, y) result(r)
    real(c_double), intent(in) :: x, y
    integer(c_int64_t) :: bx, by
    logical :: r
    integer(c_int64_t), parameter :: zmask = -huge(0_c_int64_t) - 1_c_int64_t
    bx = transfer(x, bx)
    by = transfer(y, by)
    r = (bx == by)
    if (.not. r .and. iand(bx, zmask) == 0_c_int64_t &
        .and. iand(by, zmask) == 0_c_int64_t) r = .true.
  end function amwi_equal
end module amwi_mod

subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(C, name='argmax_with_index_fp64')
  use iso_c_binding
  use omp_lib
  use amwi_mod, only: BS, bm, amwi_grow, amwi_equal
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value, intent(in) :: out_index
  type(c_ptr), value, intent(in) :: out_value
  integer(c_int64_t), value, intent(in) :: len_1d

  real(c_double), dimension(:), pointer :: av
  real(c_double), pointer :: ov
  integer(c_int64_t), pointer :: oi
  real(c_double) :: M, bmax
  integer(c_int64_t) :: n, k, i, base, hi, nblocks
  logical :: hit

  n = len_1d
  if (n <= 0_c_int64_t) then
    call c_f_pointer(out_value, ov)
    call c_f_pointer(out_index, oi)
    ov = 0.0d0
    oi = 0_c_int64_t
    return
  end if

  call c_f_pointer(a, av, [n])
  call c_f_pointer(out_value, ov)
  call c_f_pointer(out_index, oi)

  if (n == 1_c_int64_t) then
    ov = av(1)
    oi = 1_c_int64_t
    return
  end if

  nblocks = (n + BS - 1_c_int64_t) / BS
  call amwi_grow(nblocks)

  !$omp parallel do schedule(static)
  do k = 1_c_int64_t, nblocks
    base = (k - 1_c_int64_t) * BS
    hi = min(base + BS, n)
    bmax = -huge(0.0d0)
    do i = base + 1_c_int64_t, hi
      bmax = max(bmax, av(i))
    end do
    bm(k) = bmax
  end do
  !$omp end parallel do

  M = bm(1_c_int64_t)
  do k = 2_c_int64_t, nblocks
    M = max(M, bm(k))
  end do

  hit = .false.
  do k = 1_c_int64_t, nblocks
    if (amwi_equal(bm(k), M)) then
      hit = .true.
      exit
    end if
  end do

  if (hit) then
    base = (k - 1_c_int64_t) * BS
    do i = 1_c_int64_t, n - base
      if (amwi_equal(av(base + i), M)) then
        ov = M
        oi = base + i
        return
      end if
    end do
  end if

  ! Defensive fallback (only reachable with all-NaN data): exact reference semantics.
  M = av(1)
  k = 1_c_int64_t
  do i = 2_c_int64_t, n
    if (av(i) > M) then
      M = av(i)
      k = i
    end if
  end do
  ov = M
  oi = k
end subroutine argmax_with_index_fp64
