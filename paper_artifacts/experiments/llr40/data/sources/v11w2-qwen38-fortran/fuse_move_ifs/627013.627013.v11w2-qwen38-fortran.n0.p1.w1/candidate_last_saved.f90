subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in), value :: K
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: a(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(inout) :: b(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: cond(0:LEN_2D-1)
  real(c_double), intent(in) :: src(0:LEN_2D*LEN_2D-1)

  integer(c_int64_t) :: n, i, j, base
  logical :: go_a
  integer :: n4

  n = LEN_2D
  n4 = int(n, 4)

  ! fast path: small problem, single thread, let -O3 vectorize
  if (n <= 256) then
     do i = 0, n4 - 1
        base = i * n
        go_a = cond(i) > 0.0d0
        if (go_a) then
           do j = 0, n4 - 1
              a(base + j) = src(base + j) * 2.0d0
           end do
        end if
        if (K > 0) then
           do j = 0, n4 - 1
              b(base + j) = src(base + j) + 1.0d0
           end do
        end if
     end do
     return
  end if

  ! large path: per row, read src once, write b always, a when cond(i) > 0
  !$omp parallel do schedule(static) private(base, go_a)
  do i = 0, n4 - 1
     base = i * n
     go_a = cond(i) > 0.0d0
     if (go_a) then
        do j = 0, n - 1
           a(base + j) = src(base + j) * 2.0d0
        end do
     end if
     if (K > 0) then
        do j = 0, n - 1
           b(base + j) = src(base + j) + 1.0d0
        end do
     end if
  end do
  !$omp end parallel do
end subroutine fuse_move_ifs_fp64
