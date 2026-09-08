subroutine scan_affine_decay_fp64(c, x, y, n, workspace, workspace_bytes) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_bytes
  real(c_double), intent(in) :: c(n), x(n)
  real(c_double), intent(inout) :: y(n)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), parameter :: bs = 1024
  integer(c_int64_t) :: m, b, i, s, e, l
  real(c_double) :: A, B, acc, y0
  real(c_double), allocatable, target :: ab_alloc(:,:)
  real(c_double), pointer :: ab(:,:)

  if (n <= 0) return
  if (n <= 4096) then
     y(1) = x(1)
     do i = 2, n
        y(i) = c(i) * y(i-1) + x(i)
     end do
     return
  end if

  m = (n + bs - 1) / bs

  if (workspace_bytes >= 16 * m) then
     call c_f_pointer(workspace, ab, [2_c_int, int(m, c_int)])
  else
     allocate(ab_alloc(2, m))
     ab => ab_alloc
  end if

  ! First pass: local affine coefficients per block.
  !$omp parallel do schedule(static) private(b, s, e, l, i, A, B)
  do b = 1, m
     s = (b - 1) * bs + 1
     e = min(b * bs, n)
     l = s
     if (l < 2) l = 2
     if (l > e) then
        A = 1.0_c_double
        B = 0.0_c_double
     else
        A = c(l)
        B = x(l)
        do i = l + 1, e
           A = A * c(i)
           B = c(i) * B + x(i)
        end do
     end if
     ab(1, b) = A
     ab(2, b) = B
  end do

  ! Prefix combine block transforms.
  do b = 2, m
     A = ab(1, b)
     B = ab(2, b)
     ab(1, b) = A * ab(1, b - 1)
     ab(2, b) = A * ab(2, b - 1) + B
  end do

  ! Second pass: write y using global block prefixes.
  y0 = x(1)
  !$omp parallel do schedule(static) private(b, s, e, l, i, acc)
  do b = 1, m
     s = (b - 1) * bs + 1
     e = min(b * bs, n)
     if (b == 1) then
        y(s) = x(s)
        acc = y(s)
        l = s + 1
     else
        acc = ab(1, b - 1) * y0 + ab(2, b - 1)
        l = s
     end if
     do i = l, e
        acc = c(i) * acc + x(i)
        y(i) = acc
     end do
  end do

  if (allocated(ab_alloc)) deallocate(ab_alloc)
end subroutine scan_affine_decay_fp64
